import React from 'react';
import { Header } from './Header';
import { Navigation } from './Navigation';

interface MainLayoutProps {
  children: React.ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <Header />
      <Navigation />

      <main className="flex-1 container mx-auto px-4 py-6 md:px-6 lg:px-8 max-w-7xl">
        {children}
      </main>

      <footer className="border-t border-gray-200 bg-white">
        <div className="container mx-auto px-4 py-4 md:px-6 lg:px-8 max-w-7xl">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <p>&copy; 2024 GPU Config Recommender</p>
            <div className="flex gap-4">
              <a href="#" className="hover:text-gray-900">
                Help
              </a>
              <a href="#" className="hover:text-gray-900">
                About
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};
