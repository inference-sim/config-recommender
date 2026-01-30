import React from 'react';

type MetricColor = 'blue' | 'red' | 'purple' | 'orange' | 'green';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  color?: MetricColor;
  description?: string;
}

const colorMap: Record<MetricColor, { bg: string; text: string }> = {
  blue: { bg: 'bg-blue-50', text: 'text-blue-600' },
  red: { bg: 'bg-red-50', text: 'text-red-600' },
  purple: { bg: 'bg-purple-50', text: 'text-purple-600' },
  orange: { bg: 'bg-orange-50', text: 'text-orange-600' },
  green: { bg: 'bg-green-50', text: 'text-green-600' },
};

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit = '',
  color = 'blue',
  description,
}) => {
  const { bg, text } = colorMap[color];

  return (
    <div className={`${bg} rounded-lg p-4 space-y-2`}>
      <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">{label}</p>

      <div className="flex items-baseline gap-1">
        <span className={`text-2xl font-bold ${text}`}>{value}</span>
        {unit && <span className="text-sm text-gray-600">{unit}</span>}
      </div>

      {description && <p className="text-xs text-gray-600">{description}</p>}
    </div>
  );
};
