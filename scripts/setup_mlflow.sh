#!/bin/bash
################################################################################
# MLflow Server Setup Script
################################################################################

set -e
GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

echo "========================================================================"
echo "MLflow Server Setup"
echo "========================================================================"

# Configuration
MLFLOW_PORT=5000
BACKEND_STORE="postgresql://airflow:airflow123@localhost:5432/airflow"
ARTIFACT_ROOT="./mlartifacts"

log_info "Creating MLflow artifacts directory..."
mkdir -p mlartifacts

log_info "Starting MLflow server..."
echo ""
echo "Backend Store: ${BACKEND_STORE}"
echo "Artifact Root: ${ARTIFACT_ROOT}"
echo "Port: ${MLFLOW_PORT}"
echo ""

# Start MLflow server
mlflow server \
    --backend-store-uri "${BACKEND_STORE}" \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --host 0.0.0.0 \
    --port ${MLFLOW_PORT} &

MLFLOW_PID=$!

sleep 3

if ps -p $MLFLOW_PID > /dev/null; then
    log_success "MLflow server started! (PID: ${MLFLOW_PID})"
    echo ""
    echo "🌐 Access MLflow UI:"
    echo "   http://localhost:${MLFLOW_PORT}"
    echo ""
    echo "Stop server: kill ${MLFLOW_PID}"
    echo "Or use: pkill -f 'mlflow server'"
else
    echo "Failed to start MLflow server"
    exit 1
fi
