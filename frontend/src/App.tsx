import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './components/layout/MainLayout';
import { Dashboard } from './components/pages/Dashboard';
import { RecommendPage } from './components/pages/RecommendPage';
import { ModelsPage } from './components/pages/ModelsPage';
import { GPUsPage } from './components/pages/GPUsPage';
import { ComparePage } from './components/pages/ComparePage';

function App() {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/recommend" element={<RecommendPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/gpus" element={<GPUsPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="*" element={<Dashboard />} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  );
}

export default App;
