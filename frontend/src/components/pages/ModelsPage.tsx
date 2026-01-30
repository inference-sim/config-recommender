import React from 'react';
import { Card, CardBody } from '../common/Card';
import { Button } from '../common/Button';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import type { AddedModel } from '@/types';
import { Trash2 } from 'lucide-react';

export const ModelsPage: React.FC = () => {
  const [models, setModels] = useLocalStorage<AddedModel[]>('models', []);

  const handleRemove = (id: string) => {
    setModels(models.filter((m) => m.id !== id));
  };

  const handleClearAll = () => {
    if (window.confirm('Are you sure you want to clear all models?')) {
      setModels([]);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Model Management</h1>
          <p className="text-gray-600">Manage and organize your ML models</p>
        </div>
        {models.length > 0 && (
          <Button variant="destructive" onClick={handleClearAll}>
            Clear All
          </Button>
        )}
      </div>

      {models.length === 0 ? (
        <Card>
          <CardBody className="text-center py-12">
            <p className="text-gray-600 mb-4">No models added yet</p>
            <p className="text-sm text-gray-500">
              Add models from the Recommend page to get started
            </p>
          </CardBody>
        </Card>
      ) : (
        <div className="space-y-4">
          {models.map((model) => (
            <Card key={model.id}>
              <CardBody>
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-lg text-gray-900">{model.name}</h3>
                    <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-gray-600">Parameters</p>
                        <p className="font-medium">
                          {(model.num_parameters / 1e9).toFixed(1)}B
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-600">Max Sequence</p>
                        <p className="font-medium">{model.max_sequence_length}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Layers</p>
                        <p className="font-medium">{model.num_layers}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Hidden Size</p>
                        <p className="font-medium">{model.hidden_size}</p>
                      </div>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemove(model.id)}
                    icon={<Trash2 size={16} />}
                  >
                    Remove
                  </Button>
                </div>
              </CardBody>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};
