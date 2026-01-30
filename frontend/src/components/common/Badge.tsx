import React from 'react';

type BadgeColor = 'green' | 'orange' | 'red' | 'blue' | 'gray';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  color?: BadgeColor;
  children: React.ReactNode;
}

const colorClasses: Record<BadgeColor, string> = {
  green: 'bg-green-100 text-green-800',
  orange: 'bg-orange-100 text-orange-800',
  red: 'bg-red-100 text-red-800',
  blue: 'bg-blue-100 text-blue-800',
  gray: 'bg-gray-100 text-gray-800',
};

export const Badge: React.FC<BadgeProps> = ({
  color = 'gray',
  children,
  className = '',
  ...props
}) => {
  return (
    <span
      className={`
        inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
        ${colorClasses[color]} ${className}
      `}
      {...props}
    >
      {children}
    </span>
  );
};
