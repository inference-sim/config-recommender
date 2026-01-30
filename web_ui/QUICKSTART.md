# Quick Start Guide

Get the new GPU Recommendation UI running in 60 seconds!

## Prerequisites

- Python 3.8+ installed
- FastAPI and Uvicorn installed (or install via requirements.txt)

## Step 1: Install Dependencies (if needed)

```bash
pip install fastapi uvicorn
```

Or from the project root:

```bash
pip install -r requirements.txt
```

## Step 2: Start the Backend

```bash
cd web_ui
python backend.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 3: Open the UI

Open your browser and navigate to:
```
http://localhost:8000
```

## Step 4: Use the Application

### Add a Model
1. Type a model name (e.g., `Qwen/Qwen2.5-7B`)
2. Click "Add" or press Ctrl/Cmd+Enter

### Select GPUs
1. Click on GPU cards to select them
2. Selected GPUs will be highlighted

### Generate Recommendations
1. Optionally adjust advanced settings
2. Click "Generate Recommendations"
3. Results appear below

### Export Results
1. Click "JSON" or "CSV" to download results

## Tips

- **Dark Mode**: Click the moon/sun icon in the top-right
- **Persistence**: Your selections are saved automatically
- **Keyboard**: Use Ctrl/Cmd+Enter to quickly add models
- **Mobile**: The UI works great on phones and tablets

## Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process if needed
kill -9 <PID>

# Try a different port
uvicorn backend:app --port 8080
```

### UI shows "Backend not running"
1. Ensure backend is running on port 8000
2. Check browser console for errors
3. Try refreshing the page

### CORS errors
- Backend includes CORS middleware
- Should work out of the box
- Check backend logs for details

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [DESIGN_COMPARISON.md](DESIGN_COMPARISON.md) for design rationale
- Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details

## Need Help?

1. Check the troubleshooting section above
2. Review the full README.md
3. Check backend logs: `tail -f /tmp/backend.log`
4. Open an issue on GitHub

---

**Enjoy your new GPU Recommendation UI! 🚀**