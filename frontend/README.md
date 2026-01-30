# GPU Config Recommender - Frontend

A modern, production-ready React + TypeScript frontend for the GPU Config Recommender application. This interface provides an intuitive way to find optimal GPU configurations for ML model inference.

## Features

- **Modern UI/UX**: Clean, professional design following the design system specifications
- **Responsive Design**: Mobile-friendly interface that works on all devices
- **Model Management**: Add and manage ML models from HuggingFace
- **GPU Library**: Browse and select from preloaded GPU configurations
- **Recommendation Engine**: Generate GPU recommendations with performance metrics
- **Advanced Options**: Fine-tune estimation parameters (precision, memory overhead, latency bounds)
- **Data Visualization**: Charts showing memory breakdown and performance metrics
- **Export Functionality**: Download recommendations as JSON

## Technology Stack

- **React 18** - Modern React with hooks
- **TypeScript 5** - Full type safety
- **Vite** - Fast build tool and dev server
- **Tailwind CSS 3** - Utility-first styling
- **React Router 6** - Client-side routing
- **Recharts** - Data visualization
- **Axios** - HTTP client for API calls
- **Lucide React** - Modern icon library

## Prerequisites

- Node.js 18+ or npm/pnpm
- Backend API running (see main project README)

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
pnpm install
```

3. Create environment file:
```bash
cp .env.example .env
```

4. Configure the API endpoint in `.env`:
```
VITE_API_URL=http://localhost:8000/api
```

## Development

Start the development server:

```bash
npm run dev
# or
pnpm dev
```

The application will be available at `http://localhost:3000`

## Build for Production

Build the application:

```bash
npm run build
# or
pnpm build
```

Preview the production build:

```bash
npm run preview
# or
pnpm preview
```

The build output will be in the `dist/` directory.

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── common/          # Reusable UI components
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── Badge.tsx
│   │   │   └── Modal.tsx
│   │   ├── layout/          # Layout components
│   │   │   ├── Header.tsx
│   │   │   ├── Navigation.tsx
│   │   │   └── MainLayout.tsx
│   │   ├── forms/           # Form components
│   │   │   ├── Input.tsx
│   │   │   ├── Slider.tsx
│   │   │   └── AdvancedOptions.tsx
│   │   ├── results/         # Result display components
│   │   │   ├── RecommendationCard.tsx
│   │   │   └── MetricCard.tsx
│   │   ├── visualizations/  # Chart components
│   │   │   └── MemoryBreakdown.tsx
│   │   └── pages/           # Page components
│   │       ├── Dashboard.tsx
│   │       ├── RecommendPage.tsx
│   │       ├── ModelsPage.tsx
│   │       ├── GPUsPage.tsx
│   │       └── ComparePage.tsx
│   ├── hooks/               # Custom React hooks
│   │   └── useLocalStorage.ts
│   ├── services/            # API services
│   │   ├── api.ts
│   │   └── recommendationService.ts
│   ├── types/               # TypeScript types
│   │   └── index.ts
│   ├── utils/               # Utility functions
│   │   └── constants.ts
│   ├── styles/              # Global styles
│   │   └── index.css
│   ├── App.tsx              # Main app component
│   └── main.tsx             # Entry point
├── public/                  # Static assets
├── index.html               # HTML template
├── package.json             # Dependencies
├── tsconfig.json            # TypeScript config
├── vite.config.ts           # Vite config
├── tailwind.config.js       # Tailwind config
└── README.md                # This file
```

## Usage Guide

### 1. Dashboard

The landing page provides an overview and quick access to key features:
- Getting started guide
- Quick links to model and GPU management
- Information about how the tool works

### 2. Recommend Page

Main workflow for generating GPU recommendations:

1. **Add Models**: Enter HuggingFace model identifiers (e.g., `Qwen/Qwen2.5-7B`)
2. **Select GPUs**: Choose from the preloaded GPU library
3. **Configure Parameters** (optional): Adjust precision, memory overhead, latency bounds
4. **Generate**: Click "Generate Recommendations" to get results
5. **Review Results**: View recommended GPU with performance metrics
6. **Export**: Download recommendations as JSON

### 3. Models Page

Manage your model collection:
- View all added models with their specifications
- Remove individual models
- Clear all models

### 4. GPUs Page

Browse the GPU library:
- View all available GPU configurations
- See detailed specifications (memory, bandwidth, TFLOPS, cost)
- Compare different GPU options

### 5. Compare Page

Side-by-side comparison (coming soon):
- Compare multiple model/GPU combinations
- Analyze performance trade-offs

## API Integration

The frontend communicates with the backend API at `/api/recommendations`:

**Request Format:**
```json
{
  "models": [{"name": "Qwen/Qwen2.5-7B"}],
  "gpus": [{
    "name": "NVIDIA H100 80GB",
    "memory_gb": 80,
    "memory_bandwidth_gb_s": 2039,
    "tflops_fp16": 312,
    "tflops_fp32": 156,
    "cost_per_hour": 2.38
  }],
  "precision_bytes": 2,
  "memory_overhead_factor": 1.2,
  "latency_bound_ms": null,
  "input_length": null,
  "output_length": null
}
```

**Response Format:**
```json
{
  "recommendations": [{
    "model_name": "Qwen/Qwen2.5-7B",
    "recommended_gpu": "NVIDIA H100 80GB",
    "performance": {
      "tokens_per_second": 1240,
      "intertoken_latency_ms": 0.81,
      "memory_required_gb": 18.5,
      "memory_weights_gb": 13.2,
      "memory_kv_cache_gb": 5.3,
      "fits_in_memory": true,
      "tensor_parallel_size": 1
    },
    "all_compatible_gpus": [...],
    "reasoning": "..."
  }]
}
```

## State Management

The application uses:
- **React Context** for global state (if needed in future)
- **localStorage** for persisting models and GPU selections
- **Component state** for local UI state

Data is persisted in the browser's localStorage:
- `models`: Array of added models
- `gpus`: Array of selected GPUs

## Styling

The application uses Tailwind CSS with a custom design system:

**Color Palette:**
- Primary Blue: `#2563EB`
- Success Green: `#10B981`
- Warning Orange: `#F59E0B`
- Error Red: `#EF4444`
- Neutral Gray: `#6B7280`

**Component Variants:**
- Buttons: primary, secondary, ghost, destructive
- Cards: elevated, standard
- Badges: green, orange, red, blue, gray

## Code Quality

The project includes:
- **TypeScript**: Full type safety throughout
- **ESLint**: Code linting with recommended rules
- **Type checking**: `npm run type-check`

## Deployment

### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

### Deploy to Netlify

```bash
npm run build
netlify deploy --prod --dir=dist
```

### Deploy to AWS S3 + CloudFront

```bash
npm run build
aws s3 sync dist/ s3://your-bucket-name/
```

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Performance

The application is optimized for performance:
- Code splitting with React.lazy (can be added for route-based splitting)
- Optimized bundle size with Vite
- Lazy loading of heavy components
- Efficient re-renders with React.memo

## Accessibility

The UI follows WCAG 2.1 AA guidelines:
- Semantic HTML
- ARIA labels where appropriate
- Keyboard navigation support
- Focus indicators
- Color contrast ratios >= 4.5:1

## Troubleshooting

### Port 3000 already in use

Change the port in `vite.config.ts`:
```typescript
server: {
  port: 3001,
}
```

### API connection errors

1. Verify the backend is running at `http://localhost:8000`
2. Check CORS settings in the backend
3. Verify the `VITE_API_URL` in `.env`

### Build errors

1. Clear node_modules and reinstall:
```bash
rm -rf node_modules package-lock.json
npm install
```

2. Clear Vite cache:
```bash
rm -rf node_modules/.vite
```

## Contributing

When contributing to the frontend:

1. Follow the existing code structure
2. Use TypeScript for all new components
3. Add proper type definitions
4. Use Tailwind CSS for styling
5. Ensure responsive design
6. Test on multiple browsers

## License

Same as the main project (see root LICENSE file)

## Support

For issues or questions:
- Check the main project documentation
- Review the design system specifications in `DESIGN_SYSTEM.md`
- See component templates in `COMPONENT_CODE_TEMPLATES.md`

## Roadmap

Future enhancements:
- [ ] Dark mode support
- [ ] Advanced comparison views
- [ ] Real-time collaboration
- [ ] Save/load sessions
- [ ] Export to PDF
- [ ] More chart types
- [ ] Model search from HuggingFace API
- [ ] Custom GPU creation form
- [ ] Bulk operations
- [ ] Keyboard shortcuts
