import React from 'react';
import { Card, CardBody } from '../common/Card';
import { GPU_LIBRARY } from '@/utils/constants';

export const GPUsPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">GPU Library</h1>
        <p className="text-gray-600">Browse available GPU configurations</p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {GPU_LIBRARY.map((gpu) => (
          <Card key={gpu.name}>
            <CardBody>
              <h3 className="font-semibold text-lg text-gray-900 mb-3">{gpu.name}</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Memory</span>
                  <span className="font-medium">{gpu.memory_gb}GB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Bandwidth</span>
                  <span className="font-medium">{gpu.memory_bandwidth_gb_s} GB/s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">TFLOPS (FP16)</span>
                  <span className="font-medium">{gpu.tflops_fp16}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">TFLOPS (FP32)</span>
                  <span className="font-medium">{gpu.tflops_fp32}</span>
                </div>
                {gpu.cost_per_hour && (
                  <div className="flex justify-between pt-2 border-t">
                    <span className="text-gray-600">Cost</span>
                    <span className="font-medium text-blue-600">${gpu.cost_per_hour}/hr</span>
                  </div>
                )}
              </div>
            </CardBody>
          </Card>
        ))}
      </div>
    </div>
  );
};
