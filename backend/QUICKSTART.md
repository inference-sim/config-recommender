# Quick Start Guide

Get the GPU Config Recommender API running in under 5 minutes!

## Option 1: Local Development (Fastest)

1. **Install the config_recommender library:**
   ```bash
   cd /Users/jchen/go/src/llm-d/config-recommender
   pip install -e .
   ```

2. **Install backend dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Start the server:**
   ```bash
   ./start.sh --reload
   ```

   Or manually:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Test it:**
   - Open http://localhost:8000/docs in your browser
   - Or run the test script: `python test_api.py`

## Option 2: Docker (Most Portable)

1. **Build and run with docker-compose:**
   ```bash
   cd backend
   docker-compose up --build
   ```

2. **Or build and run manually:**
   ```bash
   # From project root
   docker build -f backend/Dockerfile -t gpu-config-backend .
   docker run -p 8000:8000 gpu-config-backend
   ```

3. **Test it:**
   - Open http://localhost:8000/docs in your browser
   - Or run: `python test_api.py`

## Verify Installation

Run the test suite:
```bash
python test_api.py
```

Expected output:
```
======================================================================
  Test Summary
======================================================================
✓ Health Check
✓ List GPUs
✓ Get GPU Specs
✓ Validate Model
✓ GPU Recommendation

Passed: 5/5

🎉 All tests passed!
```

## Try Your First Request

```bash
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "model": {"name": "Qwen/Qwen2.5-7B"},
    "available_gpus": [{
      "name": "NVIDIA H100 80GB",
      "memory_gb": 80.0,
      "memory_bandwidth_gb_s": 3350.0,
      "tflops_fp16": 1979.0,
      "tflops_fp32": 989.0,
      "cost_per_hour": 4.76
    }]
  }'
```

## Next Steps

- Explore the API docs at http://localhost:8000/docs
- Read the full README.md for detailed documentation
- Integrate with your frontend application

## Troubleshooting

**"Module 'config_recommender' not found"**
- Install the parent package: `pip install -e ..` from backend directory

**"Model validation takes too long"**
- First request downloads model config from HuggingFace (can take 10-30s)
- Subsequent requests are faster

**"Gated model access denied"**
- Set HF_TOKEN environment variable: `export HF_TOKEN=your_token`

## Support

For issues, check the README.md or the main project documentation.
