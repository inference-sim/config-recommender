import React, { useState } from 'react';
import { Card, CardBody } from '../common/Card';
import { Button } from '../common/Button';
import { Input } from '../forms/Input';
import { AdvancedOptions } from '../forms/AdvancedOptions';
import { RecommendationCard } from '../results/RecommendationCard';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { recommendationService } from '@/services/recommendationService';
import { GPU_LIBRARY, DEFAULT_PARAMETERS } from '@/utils/constants';
import type { AddedModel, AddedGPU, EstimationParameters, RecommendationResult } from '@/types';
import { X, Plus } from 'lucide-react';

export const RecommendPage: React.FC = () => {
  const [models, setModels] = useLocalStorage<AddedModel[]>('models', []);
  const [selectedGPUs, setSelectedGPUs] = useLocalStorage<AddedGPU[]>('gpus', []);
  const [parameters, setParameters] = useState<EstimationParameters>(DEFAULT_PARAMETERS);
  const [recommendations, setRecommendations] = useState<RecommendationResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newModelName, setNewModelName] = useState('');

  const handleAddModel = () => {
    if (!newModelName.trim()) return;

    const newModel: AddedModel = {
      id: crypto.randomUUID(),
      name: newModelName,
      num_parameters: 7000000000,
      num_layers: 32,
      hidden_size: 4096,
      num_attention_heads: 32,
      vocab_size: 32000,
      max_sequence_length: 4096,
      addedAt: new Date(),
      fetchedFrom: 'manual',
    };

    setModels([...models, newModel]);
    setNewModelName('');
  };

  const handleRemoveModel = (id: string) => {
    setModels(models.filter((m) => m.id !== id));
  };

  const handleToggleGPU = (gpu: typeof GPU_LIBRARY[0]) => {
    const exists = selectedGPUs.find((g) => g.name === gpu.name);
    if (exists) {
      setSelectedGPUs(selectedGPUs.filter((g) => g.name !== gpu.name));
    } else {
      const newGPU: AddedGPU = {
        ...gpu,
        id: crypto.randomUUID(),
        addedAt: new Date(),
        source: 'library',
      };
      setSelectedGPUs([...selectedGPUs, newGPU]);
    }
  };

  const handleGenerate = async () => {
    if (models.length === 0 || selectedGPUs.length === 0) {
      setError('Please add at least one model and one GPU');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await recommendationService.generateRecommendations(
        models,
        selectedGPUs,
        parameters
      );
      setRecommendations(response.recommendations);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate recommendations');
    } finally {
      setIsLoading(false);
    }
  };

  const handleExport = (recommendation: RecommendationResult) => {
    const blob = recommendationService.exportJSON(recommendation);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `recommendation-${recommendation.model_name}.json`;
    a.click();
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">GPU Recommendations</h1>
        <p className="text-gray-600">Generate optimal GPU configurations for your models</p>
      </div>

      <Card className="sticky top-20 z-10 bg-white">
        <CardBody className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Models Section */}
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-3">
                Selected Models ({models.length})
              </h3>
              <div className="space-y-2 mb-3">
                {models.map((model) => (
                  <div
                    key={model.id}
                    className="flex items-center justify-between bg-blue-50 p-2 rounded"
                  >
                    <span className="text-sm font-medium">{model.name}</span>
                    <button
                      onClick={() => handleRemoveModel(model.id)}
                      className="text-gray-500 hover:text-gray-700"
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <Input
                  placeholder="e.g., Qwen/Qwen2.5-7B"
                  value={newModelName}
                  onChange={(e) => setNewModelName(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleAddModel()}
                />
                <Button onClick={handleAddModel} icon={<Plus size={16} />}>
                  Add
                </Button>
              </div>
            </div>

            {/* GPUs Section */}
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-3">
                Selected GPUs ({selectedGPUs.length})
              </h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {GPU_LIBRARY.map((gpu) => {
                  const isSelected = selectedGPUs.some((g) => g.name === gpu.name);
                  return (
                    <label
                      key={gpu.name}
                      className={`flex items-center gap-2 p-2 rounded cursor-pointer ${
                        isSelected ? 'bg-blue-50 border-blue-300' : 'bg-white border-gray-200'
                      } border`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handleToggleGPU(gpu)}
                        className="w-4 h-4"
                      />
                      <div className="flex-1">
                        <p className="text-sm font-medium">{gpu.name}</p>
                        <p className="text-xs text-gray-600">
                          {gpu.memory_gb}GB • {gpu.tflops_fp16} TFLOPS
                        </p>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>
          </div>

          <AdvancedOptions
            parameters={parameters}
            onUpdate={(updates) => setParameters({ ...parameters, ...updates })}
          />

          <Button
            onClick={handleGenerate}
            isLoading={isLoading}
            disabled={models.length === 0 || selectedGPUs.length === 0}
            className="w-full"
            size="lg"
          >
            Generate Recommendations
          </Button>

          {error && <p className="text-sm text-red-600 text-center">{error}</p>}
        </CardBody>
      </Card>

      {/* Results Section */}
      {recommendations.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-gray-900">Results</h2>
          {recommendations.map((rec, idx) => (
            <RecommendationCard
              key={idx}
              recommendation={rec}
              onExport={() => handleExport(rec)}
            />
          ))}
        </div>
      )}
    </div>
  );
};
