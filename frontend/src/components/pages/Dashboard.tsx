import React from 'react';
import { Link } from 'react-router-dom';
import { Card, CardHeader, CardBody } from '../common/Card';
import { Button } from '../common/Button';
import { Zap, Database, Cpu } from 'lucide-react';

export const Dashboard: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="text-center py-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">GPU Config Recommender</h1>
        <p className="text-lg text-gray-600 mb-6">
          Find the optimal GPU configuration for your ML models
        </p>
        <Link to="/recommend">
          <Button size="lg" icon={<Zap size={20} />}>
            Start New Recommendation
          </Button>
        </Link>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <h3 className="font-semibold text-gray-900 flex items-center gap-2">
              <Database size={20} />
              Getting Started
            </h3>
          </CardHeader>
          <CardBody className="space-y-3">
            <div className="space-y-2 text-sm text-gray-600">
              <p className="font-medium text-gray-900">Quick Guide:</p>
              <ol className="list-decimal list-inside space-y-1">
                <li>Add models from HuggingFace</li>
                <li>Select GPUs from the library</li>
                <li>Generate recommendations</li>
              </ol>
            </div>
            <Link to="/models">
              <Button variant="secondary" size="sm">
                Manage Models
              </Button>
            </Link>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <h3 className="font-semibold text-gray-900 flex items-center gap-2">
              <Cpu size={20} />
              GPU Library
            </h3>
          </CardHeader>
          <CardBody className="space-y-3">
            <p className="text-sm text-gray-600">
              Browse and select from our preloaded GPU library including H100, A100, L40, and more.
            </p>
            <Link to="/gpus">
              <Button variant="secondary" size="sm">
                Browse GPUs
              </Button>
            </Link>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <h3 className="font-semibold text-gray-900 flex items-center gap-2">
              <Zap size={20} />
              Features
            </h3>
          </CardHeader>
          <CardBody className="space-y-2 text-sm text-gray-600">
            <ul className="space-y-1">
              <li>• Synthetic benchmark estimation</li>
              <li>• Roofline analysis</li>
              <li>• Tensor parallelism support</li>
              <li>• Cost optimization</li>
            </ul>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardBody className="text-center py-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">How It Works</h3>
          <p className="text-gray-600 max-w-2xl mx-auto">
            The GPU Config Recommender uses synthetic benchmarks and roofline analysis to estimate
            model performance across different GPU configurations. It automatically fetches model
            architecture details from HuggingFace and calculates memory requirements, throughput,
            and latency for optimal GPU selection.
          </p>
        </CardBody>
      </Card>
    </div>
  );
};
