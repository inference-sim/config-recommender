# Quick Start Guide

Get the GPU Config Recommender frontend up and running in 5 minutes.

## Prerequisites

- Node.js 18+ installed
- Backend API running at `http://localhost:8000`

## Setup Steps

### 1. Install Dependencies

```bash
cd frontend
npm install
```

or with pnpm (faster):

```bash
pnpm install
```

### 2. Configure Environment

Create a `.env` file:

```bash
cp .env.example .env
```

The default configuration should work if your backend is at `http://localhost:8000`:

```
VITE_API_URL=http://localhost:8000/api
```

### 3. Start Development Server

```bash
npm run dev
```

The app will open at `http://localhost:3000`

### 4. Try It Out

1. Visit `http://localhost:3000`
2. Click "Start New Recommendation"
3. Add a model (e.g., type "Qwen/Qwen2.5-7B" and click Add)
4. Select GPUs from the list
5. Click "Generate Recommendations"

## What You Should See

- **Dashboard**: Landing page with quick links
- **Recommend Page**: Main interface for generating recommendations
- **Models Page**: List of your added models
- **GPUs Page**: Available GPU configurations
- **Compare Page**: Placeholder for future comparison features

## Common Issues

### API Connection Error

Make sure the backend is running:
```bash
cd ..  # Go to project root
streamlit run streamlit_app.py
```

### Port 3000 Already in Use

The dev server will automatically try port 3001, 3002, etc.

Or manually change it in `vite.config.ts`:
```typescript
server: {
  port: 3001,
}
```

## Next Steps

- Read the full [README.md](./README.md) for detailed documentation
- Review [DESIGN_SYSTEM.md](../DESIGN_SYSTEM.md) for design specifications
- Check [COMPONENT_CODE_TEMPLATES.md](../COMPONENT_CODE_TEMPLATES.md) for component examples

## Development Tips

### Hot Reload

The dev server supports hot module replacement (HMR). Changes to components will update instantly without refreshing the page.

### TypeScript

All files are fully typed. VS Code will show type errors as you code.

### Tailwind CSS

Use Tailwind utility classes for styling. IntelliSense will autocomplete class names.

### Local Storage

Models and GPUs are persisted in browser localStorage. Clear browser data to reset.

## Build for Production

```bash
npm run build
```

Output will be in the `dist/` directory.

Preview the build:

```bash
npm run preview
```

## Get Help

- Check the [README.md](./README.md)
- Review the design docs in the project root
- Open an issue on GitHub
