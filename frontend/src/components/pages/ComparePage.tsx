import React from 'react';
import { Card, CardBody } from '../common/Card';

export const ComparePage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Compare Configurations</h1>
        <p className="text-gray-600">Side-by-side comparison of models and GPUs</p>
      </div>

      <Card>
        <CardBody className="text-center py-12">
          <p className="text-gray-600">Comparison feature coming soon</p>
          <p className="text-sm text-gray-500 mt-2">
            This will allow side-by-side analysis of different GPU configurations
          </p>
        </CardBody>
      </Card>
    </div>
  );
};
