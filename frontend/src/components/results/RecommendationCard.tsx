import React, { useState } from 'react';
import { Card, CardHeader, CardBody, CardFooter } from '../common/Card';
import { Badge } from '../common/Badge';
import { Button } from '../common/Button';
import { MetricCard } from './MetricCard';
import type { RecommendationResult } from '@/types';
import { ChevronDown, Download, GitCompare } from 'lucide-react';

interface RecommendationCardProps {
  recommendation: RecommendationResult;
  onExport: () => void;
}

export const RecommendationCard: React.FC<RecommendationCardProps> = ({
  recommendation,
  onExport,
}) => {
  const [expanded, setExpanded] = useState(false);
  const { model_name, recommended_gpu, performance, reasoning, all_compatible_gpus } =
    recommendation;

  if (!recommended_gpu || !performance) {
    return (
      <Card className="border-2 border-red-200 bg-red-50">
        <CardBody className="space-y-3">
          <div className="flex items-center gap-3">
            <span className="text-2xl">✕</span>
            <div>
              <h3 className="font-semibold text-red-900">{model_name}</h3>
              <p className="text-sm text-red-700">No compatible GPU found</p>
            </div>
          </div>
          <p className="text-sm text-red-600">{reasoning}</p>
        </CardBody>
      </Card>
    );
  }

  return (
    <Card className="border-2 border-blue-300 bg-blue-50">
      <CardHeader className="border-b-0 pb-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-bold text-lg text-gray-900">
              {model_name} → {recommended_gpu}
            </h3>
            <div className="flex items-center gap-2 mt-1">
              <Badge color="green">✓ Recommended</Badge>
              <Badge color={performance.fits_in_memory ? 'green' : 'red'}>
                {performance.fits_in_memory ? 'Fits in Memory' : 'OOM Warning'}
              </Badge>
              {performance.tensor_parallel_size && performance.tensor_parallel_size > 1 && (
                <Badge color="blue">TP={performance.tensor_parallel_size}</Badge>
              )}
            </div>
          </div>
        </div>
      </CardHeader>

      <CardBody className="space-y-4">
        {/* Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard
            label="Throughput"
            value={performance.tokens_per_second.toFixed(0)}
            unit="tok/s"
            color="blue"
          />
          <MetricCard
            label="Latency"
            value={performance.intertoken_latency_ms.toFixed(2)}
            unit="ms"
            color="red"
          />
          <MetricCard
            label="Memory"
            value={`${performance.memory_required_gb.toFixed(1)}`}
            unit="GB"
            color="purple"
          />
          <MetricCard
            label="TP Size"
            value={`${performance.tensor_parallel_size}`}
            unit="GPUs"
            color="orange"
          />
        </div>

        {/* Reasoning */}
        <p className="text-sm text-gray-700 italic">{reasoning}</p>

        {/* Alternatives */}
        {all_compatible_gpus.length > 0 && (
          <div className="border-t border-blue-200 pt-3">
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-2 text-sm font-semibold mb-2 hover:text-blue-700"
            >
              <span>Alternative GPUs ({all_compatible_gpus.length})</span>
              <ChevronDown
                size={16}
                className={`transition ${expanded ? 'rotate-180' : ''}`}
              />
            </button>
            {expanded && (
              <ul className="text-sm space-y-1">
                {all_compatible_gpus.slice(0, 5).map((alt, idx) => (
                  <li key={idx} className="text-gray-600">
                    • {alt.gpu_name} – {alt.tokens_per_second?.toFixed(0) || 'N/A'} tok/s |{' '}
                    {alt.memory_required_gb?.toFixed(1) || 'N/A'}GB
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </CardBody>

      <CardFooter>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setExpanded(!expanded)}
          icon={<ChevronDown size={16} className={`transition ${expanded ? 'rotate-180' : ''}`} />}
        >
          {expanded ? 'Hide' : 'Show'} Details
        </Button>
        <Button variant="ghost" size="sm" icon={<GitCompare size={16} />}>
          Compare
        </Button>
        <Button variant="secondary" size="sm" onClick={onExport} icon={<Download size={16} />}>
          Export
        </Button>
      </CardFooter>
    </Card>
  );
};
