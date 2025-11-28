#!/bin/bash
################################################################################
# DagsHub Integration Setup
################################################################################

set -e
GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

echo "========================================================================"
echo "DagsHub Integration Setup"
echo "========================================================================"

echo ""
echo "Enter your DagsHub credentials:"
read -p "DagsHub Username: " DAGSHUB_USER
read -sp "DagsHub Token: " DAGSHUB_TOKEN
echo ""

DAGSHUB_REPO="${DAGSHUB_USER}/sales-forecasting-mlops"

log_info "Configuring MLflow tracking..."

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="https://dagshub.com/${DAGSHUB_REPO}.mlflow"
export MLFLOW_TRACKING_USERNAME="${DAGSHUB_USER}"
export MLFLOW_TRACKING_PASSWORD="${DAGSHUB_TOKEN}"

# Update .env file
if [[ -f ".env" ]]; then
    sed -i "/^MLFLOW_TRACKING_URI=/d" .env
    sed -i "/^MLFLOW_TRACKING_USERNAME=/d" .env
    sed -i "/^MLFLOW_TRACKING_PASSWORD=/d" .env
    
    {
        echo "MLFLOW_TRACKING_URI=https://dagshub.com/${DAGSHUB_REPO}.mlflow"
        echo "MLFLOW_TRACKING_USERNAME=${DAGSHUB_USER}"
        echo "MLFLOW_TRACKING_PASSWORD=${DAGSHUB_TOKEN}"
    } >> .env
fi

log_success "DagsHub configured!"

echo ""
echo "📊 Configuration:"
echo "   MLflow URI: https://dagshub.com/${DAGSHUB_REPO}.mlflow"
echo "   Username: ${DAGSHUB_USER}"
echo ""
echo "🔗 Access your DagsHub repo:"
echo "   https://dagshub.com/${DAGSHUB_REPO}"
echo ""

log_success "Setup complete!"
