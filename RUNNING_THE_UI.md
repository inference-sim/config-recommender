# Running the Production UI

This guide shows you how to run the new production-ready web UI for the GPU Config Recommender.

## 🚀 Quick Start (2 Steps)

### Step 1: Start the Backend

```bash
# From the project root directory
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will start at: **http://localhost:8000**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/health

### Step 2: Start the Frontend (in a new terminal)

```bash
# From the project root directory
cd frontend
npm run dev
```

The frontend will start at: **http://localhost:3000**

### Step 3: Open Your Browser

Visit **http://localhost:3000** and start using the UI!

---

## 📋 Detailed Instructions

### First Time Setup

#### Backend Setup
```bash
# From the project root directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fastapi; print('FastAPI installed successfully')"
```

#### Frontend Setup
```bash
# From the project root directory
cd frontend

# Install Node.js dependencies (only needed once)
npm install

# Verify build works
npm run build
```

---

## 🎮 Running the Services

### Backend Commands

#### Development Mode (auto-reloads on code changes):
```bash
cd backend
uvicorn app.main:app --reload
```

#### Production Mode:
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Custom Port:
```bash
cd backend
uvicorn app.main:app --reload --port 8080
```

#### With Environment Variables:
```bash
cd backend
HF_TOKEN=your_token_here uvicorn app.main:app --reload
```

### Frontend Commands

#### Development Mode (hot reload):
```bash
cd frontend
npm run dev
```

#### Production Build:
```bash
cd frontend
npm run build
npm run preview  # Preview the production build
```

#### Custom Port:
```bash
cd frontend
npm run dev -- --port 3001
```

---

## 🛑 Stopping the Services

### Stop Backend
- **If running in terminal**: Press `Ctrl+C`
- **If running in background**:
  ```bash
  pkill -9 -f "uvicorn app.main:app"
  ```

### Stop Frontend
- **If running in terminal**: Press `Ctrl+C`
- **If running in background**:
  ```bash
  pkill -9 -f "vite"
  ```

### Stop All Services
```bash
pkill -9 -f "uvicorn app.main:app"
pkill -9 -f "vite"
```

---

## 🔄 Restarting After Code Changes

### Backend Changes
1. Press `Ctrl+C` to stop the backend
2. Re-run: `uvicorn app.main:app --reload`

Or if using `--reload` flag, changes are picked up automatically!

### Frontend Changes
No restart needed! Vite hot-reloads automatically when you save files.

---

## 🐳 Alternative: Run with Docker

### Run Backend Only:
```bash
cd backend
docker build -t gpu-recommender-backend .
docker run -p 8000:8000 gpu-recommender-backend
```

### Run Both Services:
```bash
cd backend
docker-compose up --build
```

This starts both backend (port 8000) and frontend (port 3000) together.

---

## 🧪 Testing the Setup

### Test Backend is Running:
```bash
curl http://localhost:8000/api/health
# Should return: {"status":"healthy"}
```

Or visit: http://localhost:8000/docs (Swagger UI)

### Test Frontend is Running:
Open http://localhost:3000 in your browser. You should see the GPU Config Recommender dashboard.

### Full Integration Test:
```bash
cd backend
python test_api.py
```

---

## 📖 Using the UI

### Basic Workflow:

1. **Add Models**
   - Go to "Recommend" page
   - Enter a HuggingFace model name (e.g., `Qwen/Qwen2.5-7B`)
   - Click "Add Model"

2. **Select GPUs**
   - Check the GPUs you want to compare
   - GPUs are preloaded from the library (H100, A100, etc.)

3. **Configure Parameters**
   - Set input/output sequence lengths
   - Adjust memory overhead factor
   - Set latency bounds (optional)

4. **Generate Recommendations**
   - Click "Generate Recommendation"
   - View results with performance metrics
   - See throughput, latency, memory usage

5. **Export Results**
   - Download as JSON or CSV
   - Copy metrics for analysis

---

## 🔧 Troubleshooting

### Problem: Backend Returns 422 Error
**Cause**: API request format mismatch
**Solution**: Make sure you're running the latest code. The API was recently fixed to handle this.

### Problem: Backend Returns `inf` Value Error
**Cause**: Model too large for GPUs, causing infinity calculations
**Solution**: This is now fixed with sanitization. Restart the backend:
```bash
pkill -9 -f "uvicorn"
cd backend
uvicorn app.main:app --reload
```

### Problem: Frontend Can't Connect to Backend
**Cause**: Backend not running or wrong port
**Solution**:
1. Check backend is running: `curl http://localhost:8000/api/health`
2. Check frontend API URL in `frontend/src/services/api.ts`
3. Make sure CORS is enabled in backend

### Problem: Module Not Found Errors (Backend)
**Cause**: Missing dependencies
**Solution**:
```bash
cd backend
pip install -r requirements.txt
```

### Problem: Module Not Found Errors (Frontend)
**Cause**: Missing npm packages
**Solution**:
```bash
cd frontend
npm install
```

### Problem: Port Already in Use
**Cause**: Another service using the port
**Solution**:
```bash
# Find what's using port 8000
lsof -i :8000
# Kill it
kill -9 <PID>

# Or use a different port
uvicorn app.main:app --reload --port 8001
```

### Problem: HuggingFace Model Not Found
**Cause**: Model is gated or doesn't exist
**Solution**:
1. Verify model name on HuggingFace.co
2. For gated models, set `HF_TOKEN` environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   uvicorn app.main:app --reload
   ```
3. Or use manual parameter overrides in the UI

### Problem: TypeScript Errors in Frontend
**Cause**: Type mismatches or outdated build
**Solution**:
```bash
cd frontend
npm run type-check  # Check for errors
rm -rf node_modules package-lock.json
npm install  # Reinstall dependencies
```

---

## 🌐 Environment Variables

### Backend (`backend/.env`)
```bash
# Optional: HuggingFace token for gated models
HF_TOKEN=your_huggingface_token_here

# Optional: Allowed CORS origins
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001

# Optional: API port
PORT=8000
```

### Frontend (`frontend/.env`)
```bash
# Backend API URL
VITE_API_URL=http://localhost:8000/api
```

---

## 📊 Monitoring and Logs

### View Backend Logs:
Backend logs appear in the terminal where uvicorn is running.

### View Frontend Logs:
- Browser console (F12 → Console tab)
- Terminal where `npm run dev` is running

### Enable Debug Mode:
```bash
# Backend
uvicorn app.main:app --reload --log-level debug

# Frontend (already shows verbose logs in dev mode)
npm run dev
```

---

## 🚀 Production Deployment

### Backend Production:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend Production:
```bash
cd frontend
npm run build
npm install -g serve
serve -s dist -l 3000
```

### With Nginx (Recommended):
See `backend/docker-compose.yml` for a complete production setup with Nginx reverse proxy.

---

## 📚 Additional Resources

- **Backend API Documentation**: http://localhost:8000/docs
- **Backend README**: `backend/README.md`
- **Frontend README**: `frontend/README.md`
- **Design Documentation**: See `DESIGN_SYSTEM.md` in project root
- **Implementation Guide**: See `IMPLEMENTATION_GUIDE.md` in project root

---

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review backend logs for error messages
3. Check browser console for frontend errors
4. Verify both services are running: `ps aux | grep -E "uvicorn|vite"`
5. Try restarting both services

For bugs or feature requests, create an issue in the repository.

---

## 💡 Pro Tips

### Use tmux for Multiple Services:
```bash
# Start backend in tmux
tmux new -s backend
cd backend && uvicorn app.main:app --reload
# Detach: Ctrl+B, then D

# Start frontend in tmux
tmux new -s frontend
cd frontend && npm run dev
# Detach: Ctrl+B, then D

# List sessions: tmux ls
# Attach: tmux attach -t backend
```

### Quick Restart Script:
Create `restart.sh` in project root:
```bash
#!/bin/bash
pkill -9 -f "uvicorn app.main:app"
pkill -9 -f "vite"
cd backend && uvicorn app.main:app --reload &
cd frontend && npm run dev &
echo "Services restarted!"
```

Make it executable: `chmod +x restart.sh`
Run it: `./restart.sh`

---

**Happy Recommending! 🎉**
