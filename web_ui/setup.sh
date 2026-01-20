#!/bin/bash

echo "🚀 Setting up GPU Recommendation Engine Web UI..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install config_recommender if not already installed
echo ""
echo "📦 Installing config_recommender..."
cd ..
pip install -e .
cd web_ui

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the server, run:"
echo "  python backend.py"
echo ""
echo "Then open your browser to: http://localhost:8000"

# Made with Bob
