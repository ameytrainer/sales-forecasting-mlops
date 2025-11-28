#!/bin/bash
################################################################################
# API Deployment Script
################################################################################

set -e
GREEN='\033[0;32m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "========================================================================"
echo "FastAPI Application Deployment"
echo "========================================================================"

# Load environment
if [[ -f ".env" ]]; then
    source .env
else
    log_error ".env file not found!"
    exit 1
fi

# Configuration
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
API_WORKERS="${API_WORKERS:-4}"

log_info "Deployment Configuration:"
echo "   Host: ${API_HOST}"
echo "   Port: ${API_PORT}"
echo "   Workers: ${API_WORKERS}"
echo ""

# Check if port is available
if lsof -Pi :${API_PORT} -sTCP:LISTEN -t >/dev/null ; then
    log_error "Port ${API_PORT} is already in use!"
    echo "Kill existing process? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        lsof -ti:${API_PORT} | xargs kill -9
        log_success "Killed existing process"
    else
        exit 1
    fi
fi

# Activate virtual environment
if [[ -d "venv" ]]; then
    source venv/bin/activate
else
    log_error "Virtual environment not found!"
    exit 1
fi

# Check dependencies
log_info "Checking dependencies..."
pip install -q -r api/requirements.txt

# Start API server
log_info "Starting FastAPI server..."

uvicorn api.main:app \
    --host ${API_HOST} \
    --port ${API_PORT} \
    --workers ${API_WORKERS} \
    --log-level info \
    --access-log &

API_PID=$!
sleep 3

# Verify server is running
if ps -p $API_PID > /dev/null; then
    log_success "API server started! (PID: ${API_PID})"
    echo ""
    echo "🌐 API Access:"
    echo "   http://localhost:${API_PORT}"
    echo "   Swagger UI: http://localhost:${API_PORT}/docs"
    echo "   ReDoc: http://localhost:${API_PORT}/redoc"
    echo ""
    echo "📝 Logs:"
    echo "   tail -f logs/api.log"
    echo ""
    echo "🛑 Stop: kill ${API_PID}"
    
    # Save PID
    echo ${API_PID} > /tmp/mlops-api.pid
else
    log_error "Failed to start API server!"
    exit 1
fi
