import React from 'react';
import { Settings } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="border-b border-gray-200 bg-white sticky top-0 z-40">
      <div className="container mx-auto px-4 py-4 md:px-6 lg:px-8 max-w-7xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg flex items-center justify-center">
              <span className="text-white text-lg font-bold">⚡</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">GPU Recommender</h1>
              <p className="text-xs text-gray-600">Find optimal GPU for ML models</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button
              className="hidden md:flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-lg transition"
              title="Settings"
            >
              <Settings size={18} />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
