import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { BarChart3, Zap, Database, Cpu, GitCompare } from 'lucide-react';

const navItems = [
  { path: '/', label: 'Dashboard', icon: BarChart3 },
  { path: '/recommend', label: 'Recommend', icon: Zap },
  { path: '/models', label: 'Models', icon: Database },
  { path: '/gpus', label: 'GPUs', icon: Cpu },
  { path: '/compare', label: 'Compare', icon: GitCompare },
];

export const Navigation: React.FC = () => {
  const location = useLocation();

  return (
    <nav className="border-b border-gray-200 bg-white">
      <div className="container mx-auto px-4 md:px-6 lg:px-8 max-w-7xl">
        <div className="flex gap-1 overflow-x-auto md:overflow-visible">
          {navItems.map(({ path, label, icon: Icon }) => (
            <Link
              key={path}
              to={path}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 transition ${
                location.pathname === path
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
              }`}
            >
              <Icon size={18} />
              <span className="hidden sm:inline">{label}</span>
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
};
