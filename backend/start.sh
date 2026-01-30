#!/bin/bash
# Startup script for GPU Config Recommender Backend

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting GPU Config Recommender Backend...${NC}"

# Check if we're in the backend directory
if [ ! -f "app/main.py" ]; then
    echo -e "${RED}Error: app/main.py not found. Please run this script from the backend directory.${NC}"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found. Please install Python 3.11+${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${YELLOW}Using Python ${PYTHON_VERSION}${NC}"

# Parse command line arguments
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--reload] [--port PORT] [--host HOST]"
            exit 1
            ;;
    esac
done

# Check if config_recommender is installed
python -c "import config_recommender" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: config_recommender not found. Installing parent package...${NC}"
    cd ..
    pip install -e .
    cd backend
fi

# Check if FastAPI is installed
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    pip install -r requirements.txt
fi

# Start the server
echo -e "${GREEN}Starting server at http://${HOST}:${PORT}${NC}"
echo -e "${GREEN}API docs available at http://${HOST}:${PORT}/docs${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

uvicorn app.main:app --host "$HOST" --port "$PORT" $RELOAD
