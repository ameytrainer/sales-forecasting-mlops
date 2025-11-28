#!/bin/bash

################################################################################
# DVC Setup Script
#
# Configures DVC with DagsHub remote storage
#
# Author: Amey Talkatkar
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

print_header "DVC Configuration Setup"

# Check if DVC is initialized
if [[ ! -d ".dvc" ]]; then
    log_warning "DVC not initialized. Run 'dvc init' first."
    exit 1
fi

log_info "Configuring DVC remote storage..."

# Prompt for DagsHub credentials
echo ""
echo "Enter your DagsHub credentials:"
read -p "DagsHub Username: " DAGSHUB_USER
read -sp "DagsHub Token: " DAGSHUB_TOKEN
echo ""

# Configure DagsHub remote
log_info "Setting up DagsHub remote..."

DAGSHUB_REPO="${DAGSHUB_USER}/sales-forecasting-mlops"

dvc remote add -d dagshub https://dagshub.com/${DAGSHUB_REPO}.dvc
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user ${DAGSHUB_USER}
dvc remote modify dagshub --local password ${DAGSHUB_TOKEN}

log_success "DVC remote configured!"

# Test connection
log_info "Testing DVC remote connection..."

if dvc remote list | grep -q "dagshub"; then
    log_success "DVC remote 'dagshub' added successfully"
else
    log_warning "Failed to add DVC remote"
    exit 1
fi

# Display configuration
echo ""
echo "📊 DVC Configuration:"
echo "   Remote: dagshub"
echo "   URL: https://dagshub.com/${DAGSHUB_REPO}.dvc"
echo ""

# Save configuration to .env
if [[ -f ".env" ]]; then
    if ! grep -q "DAGSHUB_USER" .env; then
        echo "# DagsHub Configuration" >> .env
        echo "DAGSHUB_USER=${DAGSHUB_USER}" >> .env
        echo "DAGSHUB_REPO=${DAGSHUB_REPO}" >> .env
    fi
    log_success "Configuration saved to .env"
fi

# Instructions
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Track a data file:"
echo "   dvc add data/raw/sales_data.csv"
echo ""
echo "2. Commit changes:"
echo "   git add data/raw/sales_data.csv.dvc data/raw/.gitignore"
echo "   git commit -m 'Track data with DVC'"
echo ""
echo "3. Push to remote:"
echo "   dvc push"
echo ""
echo "4. Pull from remote:"
echo "   dvc pull"
echo ""

log_success "DVC setup complete!"
