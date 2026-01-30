import React from 'react';
import { PieChart, Pie, Cell, Legend, Tooltip, ResponsiveContainer } from 'recharts';

interface MemoryBreakdownProps {
  weights: number;
  kvCache: number;
  available: number;
}

export const MemoryBreakdown: React.FC<MemoryBreakdownProps> = ({
  weights,
  kvCache,
  available,
}) => {
  const data = [
    { name: `Weights`, value: weights, color: '#8B5CF6' },
    { name: `KV Cache`, value: kvCache, color: '#F59E0B' },
    { name: `Available`, value: available, color: '#E5E7EB' },
  ];

  return (
    <div className="space-y-4">
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip formatter={(value: number) => `${value.toFixed(1)}GB`} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-3 gap-2 text-sm">
        <div className="bg-purple-50 p-2 rounded">
          <p className="text-purple-600 font-semibold">{weights.toFixed(1)}GB</p>
          <p className="text-purple-600 text-xs">Weights</p>
        </div>
        <div className="bg-orange-50 p-2 rounded">
          <p className="text-orange-600 font-semibold">{kvCache.toFixed(1)}GB</p>
          <p className="text-orange-600 text-xs">KV Cache</p>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <p className="text-gray-600 font-semibold">
            {((available / (weights + kvCache + available)) * 100).toFixed(0)}%
          </p>
          <p className="text-gray-600 text-xs">Free</p>
        </div>
      </div>
    </div>
  );
};
