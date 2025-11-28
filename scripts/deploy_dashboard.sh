#!/bin/bash

################################################################################
# Dashboard Deployment Script
#
# Deploys Streamlit Dashboard for MLOps monitoring
#
# Author: Amey Talkatkar
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "========================================================================"
echo "Streamlit Dashboard Deployment"
echo "========================================================================"

# Load environment
if [[ -f ".env" ]]; then
    source .env
else
    log_error ".env file not found!"
    exit 1
fi

# Configuration
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"

log_info "Deployment Configuration:"
echo "   Host: ${DASHBOARD_HOST}"
echo "   Port: ${DASHBOARD_PORT}"
echo ""

# Check if port is available
if lsof -Pi :${DASHBOARD_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_error "Port ${DASHBOARD_PORT} is already in use!"
    echo "Kill existing process? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        lsof -ti:${DASHBOARD_PORT} | xargs kill -9
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
pip install -q -r dashboard/requirements.txt

# Create Streamlit config
mkdir -p ~/.streamlit

cat > ~/.streamlit/config.toml <<EOF
[server]
port = ${DASHBOARD_PORT}
headless = true
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
EOF

log_success "Streamlit configuration created"

# Start dashboard
log_info "Starting Streamlit dashboard..."

streamlit run dashboard/app.py \
    --server.port ${DASHBOARD_PORT} \
    --server.address ${DASHBOARD_HOST} \
    --server.headless true \
    --browser.gatherUsageStats false &

DASHBOARD_PID=$!
sleep 5

# Verify dashboard is running
if ps -p $DASHBOARD_PID > /dev/null; then
    log_success "Dashboard started! (PID: ${DASHBOARD_PID})"
    echo ""
    echo "🌐 Dashboard Access:"
    echo "   http://localhost:${DASHBOARD_PORT}"
    echo ""
    echo "📊 Available Pages:"
    echo "   - Model Comparison"
    echo "   - Experiment Tracking"
    echo "   - Predictions"
    echo "   - Data Drift"
    echo "   - System Health"
    echo ""
    echo "🛑 Stop: kill ${DASHBOARD_PID}"
    
    # Save PID
    echo ${DASHBOARD_PID} > /tmp/mlops-dashboard.pid
else
    log_error "Failed to start dashboard!"
    exit 1
fi

# Display next steps
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Access dashboard at: http://localhost:${DASHBOARD_PORT}"
echo "2. Ensure API is running: http://localhost:8000"
echo "3. Check logs: streamlit logs"
echo ""

log_success "Dashboard deployment complete!"
