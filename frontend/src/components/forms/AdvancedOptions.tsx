import React, { useState } from 'react';
import { Card, CardBody } from '../common/Card';
import { Input } from './Input';
import { Slider } from './Slider';
import { Button } from '../common/Button';
import { ChevronDown } from 'lucide-react';
import type { EstimationParameters } from '@/types';

interface AdvancedOptionsProps {
  parameters: EstimationParameters;
  onUpdate: (parameters: Partial<EstimationParameters>) => void;
}

export const AdvancedOptions: React.FC<AdvancedOptionsProps> = ({ parameters, onUpdate }) => {
  const [expanded, setExpanded] = useState(false);

  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="flex items-center gap-2 text-sm text-gray-700 hover:text-gray-900 py-2"
      >
        <ChevronDown size={16} />
        <span>Show Advanced Options</span>
      </button>
    );
  }

  const handleReset = () => {
    onUpdate({
      precision: 'fp16',
      input_length: null,
      output_length: null,
      memory_overhead_factor: 1.2,
      latency_bound_ms: null,
    });
  };

  return (
    <Card className="mt-4">
      <CardBody className="space-y-6">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-900">Advanced Parameters</h3>
          <button
            onClick={() => setExpanded(false)}
            className="text-sm text-gray-600 hover:text-gray-900"
          >
            Hide
          </button>
        </div>

        {/* Precision */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">Precision</label>
          <div className="space-y-2">
            {(['fp16', 'fp32'] as const).map((precision) => (
              <label key={precision} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  checked={parameters.precision === precision}
                  onChange={() => onUpdate({ precision })}
                  className="w-4 h-4"
                />
                <span className="text-sm text-gray-700">
                  {precision.toUpperCase()} ({precision === 'fp16' ? '2' : '4'} bytes/param)
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Sequence Lengths */}
        <div className="grid grid-cols-2 gap-4">
          <Input
            label="Input Length"
            type="number"
            value={parameters.input_length ?? ''}
            onChange={(e) =>
              onUpdate({ input_length: e.target.value ? parseInt(e.target.value) : null })
            }
            placeholder="0 (default: 1)"
            helperText="Tokens for prefill phase"
          />
          <Input
            label="Output Length"
            type="number"
            value={parameters.output_length ?? ''}
            onChange={(e) =>
              onUpdate({ output_length: e.target.value ? parseInt(e.target.value) : null })
            }
            placeholder="0 (default: 1)"
            helperText="Tokens for decode phase"
          />
        </div>

        {/* Memory Overhead */}
        <Slider
          label="Memory Overhead Factor"
          min={1.0}
          max={2.0}
          step={0.05}
          value={parameters.memory_overhead_factor}
          onChange={(value) => onUpdate({ memory_overhead_factor: value })}
          unit="x"
          showValue={true}
        />

        {/* Latency Bound */}
        <Input
          label="Max Acceptable Latency"
          type="number"
          value={parameters.latency_bound_ms ?? ''}
          onChange={(e) =>
            onUpdate({ latency_bound_ms: e.target.value ? parseFloat(e.target.value) : null })
          }
          placeholder="0 (no limit)"
          helperText="ms per token"
        />

        {/* Action Buttons */}
        <div className="flex gap-2 pt-4 border-t">
          <Button variant="ghost" size="sm" onClick={handleReset}>
            Reset to Defaults
          </Button>
        </div>
      </CardBody>
    </Card>
  );
};
