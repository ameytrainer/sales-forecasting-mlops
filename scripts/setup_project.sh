#!/bin/bash

################################################################################
# Master Project Setup Script
#
# Complete automation for Sales Forecasting MLOps Pipeline
# Sets up entire project from scratch on Ubuntu 24.04
#
# Author: Amey Talkatkar
# Email: ameytalkatkar169@gmail.com
# Course: MLOps with Agentic AI - Advanced Certification
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="sales-forecasting-mlops"
PROJECT_DIR="/home/ubuntu/${PROJECT_NAME}"
VENV_DIR="${PROJECT_DIR}/venv"
PYTHON_VERSION="3.12"

################################################################################
# Utility Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 is installed"
        return 0
    else
        log_warning "$1 is not installed"
        return 1
    fi
}

################################################################################
# System Requirements Check
################################################################################

check_system_requirements() {
    print_header "Checking System Requirements"
    
    # Check OS
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        log_info "OS: $NAME $VERSION"
    else
        log_error "Cannot determine OS version"
        exit 1
    fi
    
    # Check memory
    total_mem=$(free -m | awk '/^Mem:/{print $2}')
    log_info "Total Memory: ${total_mem}MB"
    
    if [[ $total_mem -lt 1500 ]]; then
        log_warning "Memory is less than 2GB. Some operations may be slow."
    fi
    
    # Check disk space
    available_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    log_info "Available Disk Space: ${available_space}GB"
    
    if [[ $available_space -lt 10 ]]; then
        log_error "Less than 10GB disk space available. Please free up space."
        exit 1
    fi
    
    log_success "System requirements check passed"
}

################################################################################
# Install System Dependencies
################################################################################

install_system_dependencies() {
    print_header "Installing System Dependencies"
    
    log_info "Updating package list..."
    sudo apt-get update -qq
    
    log_info "Installing essential packages..."
    sudo apt-get install -y -qq \
        build-essential \
        python3-dev \
        python3-pip \
        python3-venv \
        git \
        curl \
        wget \
        vim \
        htop \
        tmux \
        postgresql \
        postgresql-contrib \
        nginx \
        ca-certificates \
        gnupg \
        lsb-release
    
    log_success "System dependencies installed"
}

################################################################################
# Setup Python Virtual Environment
################################################################################

setup_python_venv() {
    print_header "Setting Up Python Virtual Environment"
    
    # Create project directory
    if [[ ! -d "$PROJECT_DIR" ]]; then
        log_info "Creating project directory: $PROJECT_DIR"
        mkdir -p "$PROJECT_DIR"
    fi
    
    cd "$PROJECT_DIR"
    
    # Create virtual environment
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "${VENV_DIR}/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel -q
    
    log_success "Python virtual environment ready"
}

################################################################################
# Install Python Dependencies
################################################################################

install_python_dependencies() {
    print_header "Installing Python Dependencies"
    
    source "${VENV_DIR}/bin/activate"
    
    # Check if requirements.txt exists
    if [[ -f "${PROJECT_DIR}/requirements.txt" ]]; then
        log_info "Installing from requirements.txt..."
        pip install -r "${PROJECT_DIR}/requirements.txt" -q
    else
        log_warning "requirements.txt not found, installing core packages..."
        pip install -q \
            pandas==2.2.3 \
            numpy==2.3.5 \
            scikit-learn==1.6.1 \
            xgboost==2.1.3 \
            mlflow==3.6.0 \
            dvc==3.58.1 \
            fastapi==0.115.5 \
            uvicorn[standard]==0.32.1 \
            streamlit==1.40.2 \
            plotly==5.24.1 \
            apache-airflow==3.1.3 \
            great-expectations==1.2.6 \
            psycopg2-binary==2.9.10
    fi
    
    log_success "Python dependencies installed"
}

################################################################################
# Setup PostgreSQL Database
################################################################################

setup_postgresql() {
    print_header "Setting Up PostgreSQL Database"
    
    # Check if PostgreSQL is running
    if sudo systemctl is-active --quiet postgresql; then
        log_info "PostgreSQL is already running"
    else
        log_info "Starting PostgreSQL..."
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    fi
    
    # Create database and user
    log_info "Creating database and user..."
    
    sudo -u postgres psql <<EOF
-- Create user if not exists
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'airflow') THEN
        CREATE USER airflow WITH PASSWORD 'airflow123';
    END IF;
END
\$\$;

-- Create database if not exists
SELECT 'CREATE DATABASE airflow OWNER airflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
EOF
    
    log_success "PostgreSQL database configured"
}

################################################################################
# Setup Git Repository
################################################################################

setup_git_repository() {
    print_header "Setting Up Git Repository"
    
    cd "$PROJECT_DIR"
    
    if [[ ! -d ".git" ]]; then
        log_info "Initializing git repository..."
        git init
        
        # Create .gitignore
        cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# MLflow
mlruns/
mlartifacts/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Coverage
.coverage
htmlcov/
EOF
        
        git add .gitignore
        git commit -m "Initial commit: Add .gitignore"
        
        log_success "Git repository initialized"
    else
        log_info "Git repository already exists"
    fi
}

################################################################################
# Create Project Structure
################################################################################

create_project_structure() {
    print_header "Creating Project Structure"
    
    cd "$PROJECT_DIR"
    
    # Create directory structure
    directories=(
        "data/raw"
        "data/processed"
        "models"
        "logs"
        "notebooks"
        "src/data"
        "src/features"
        "src/models"
        "src/monitoring"
        "api/routes"
        "api/services"
        "dashboard/pages"
        "dashboard/components"
        "tests"
        "scripts"
        "deployment/systemd"
        "deployment/nginx"
        "deployment/docker"
        "airflow/dags"
        "airflow/plugins"
        "airflow/logs"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            touch "${dir}/.gitkeep"
        fi
    done
    
    log_success "Project structure created"
}

################################################################################
# Setup DVC
################################################################################

setup_dvc() {
    print_header "Setting Up DVC"
    
    cd "$PROJECT_DIR"
    
    if [[ ! -f ".dvc/config" ]]; then
        log_info "Initializing DVC..."
        source "${VENV_DIR}/bin/activate"
        
        dvc init
        
        log_info "Run 'bash scripts/setup_dvc.sh' to configure DVC remote"
        log_success "DVC initialized"
    else
        log_info "DVC already initialized"
    fi
}

################################################################################
# Setup MLflow
################################################################################

setup_mlflow() {
    print_header "Setting Up MLflow"
    
    log_info "Run 'bash scripts/setup_mlflow.sh' to start MLflow server"
    log_info "Run 'bash scripts/setup_dagshub.sh' to configure DagsHub"
}

################################################################################
# Setup Airflow
################################################################################

setup_airflow() {
    print_header "Setting Up Apache Airflow"
    
    export AIRFLOW_HOME="${PROJECT_DIR}/airflow"
    
    if [[ ! -f "${AIRFLOW_HOME}/airflow.cfg" ]]; then
        log_info "Initializing Airflow..."
        
        source "${VENV_DIR}/bin/activate"
        
        # Initialize database
        airflow db migrate
        
        # Create admin user
        airflow users create \
            --username admin \
            --firstname Admin \
            --lastname User \
            --role Admin \
            --email ameytalkatkar169@gmail.com \
            --password admin123
        
        log_success "Airflow initialized"
        log_info "Username: admin"
        log_info "Password: admin123"
    else
        log_info "Airflow already initialized"
    fi
}

################################################################################
# Create Environment File
################################################################################

create_env_file() {
    print_header "Creating Environment Configuration"
    
    cd "$PROJECT_DIR"
    
    if [[ ! -f ".env" ]]; then
        cat > .env <<EOF
# Project Configuration
PROJECT_NAME=${PROJECT_NAME}
PROJECT_ROOT=${PROJECT_DIR}
ENVIRONMENT=development

# Python
PYTHON_VERSION=${PYTHON_VERSION}
PYTHONPATH=${PROJECT_DIR}

# Database
DATABASE_URL=postgresql://airflow:airflow123@localhost:5432/airflow

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=sales_forecasting
MLFLOW_MODEL_NAME=sales_forecasting_production

# DVC
DVC_REMOTE=dagshub

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false

# Dashboard
DASHBOARD_PORT=8501

# Airflow
AIRFLOW_HOME=${PROJECT_DIR}/airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql://airflow:airflow123@localhost:5432/airflow
AIRFLOW__CORE__LOAD_EXAMPLES=False

# Monitoring
DRIFT_DETECTION_THRESHOLD=0.05
PERFORMANCE_DEGRADATION_THRESHOLD=0.10
EOF
        
        log_success "Environment file created: .env"
        log_warning "Please update credentials in .env file"
    else
        log_info "Environment file already exists"
    fi
}

################################################################################
# Display Next Steps
################################################################################

display_next_steps() {
    print_header "Setup Complete!"
    
    echo ""
    log_success "Project setup completed successfully!"
    echo ""
    echo "📁 Project Location: $PROJECT_DIR"
    echo "🐍 Virtual Environment: $VENV_DIR"
    echo ""
    echo "🚀 Next Steps:"
    echo ""
    echo "1. Activate virtual environment:"
    echo "   source ${VENV_DIR}/bin/activate"
    echo ""
    echo "2. Configure DVC remote:"
    echo "   bash scripts/setup_dvc.sh"
    echo ""
    echo "3. Start MLflow server:"
    echo "   bash scripts/setup_mlflow.sh"
    echo ""
    echo "4. Configure DagsHub:"
    echo "   bash scripts/setup_dagshub.sh"
    echo ""
    echo "5. Start Airflow:"
    echo "   cd airflow && airflow standalone"
    echo ""
    echo "6. Deploy API:"
    echo "   bash scripts/deploy_api.sh"
    echo ""
    echo "7. Deploy Dashboard:"
    echo "   bash scripts/deploy_dashboard.sh"
    echo ""
    echo "📖 Documentation:"
    echo "   - README.md"
    echo "   - INSTALLATION.md"
    echo "   - STUDENT_GUIDE.md"
    echo ""
}

################################################################################
# Main Execution
################################################################################

main() {
    print_header "Sales Forecasting MLOps Pipeline - Complete Setup"
    
    echo "This script will set up the complete MLOps pipeline."
    echo "Estimated time: 10-15 minutes"
    echo ""
    
    read -p "Continue? (y/n) " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warning "Setup cancelled"
        exit 0
    fi
    
    # Execute setup steps
    check_system_requirements
    install_system_dependencies
    setup_python_venv
    install_python_dependencies
    setup_postgresql
    setup_git_repository
    create_project_structure
    setup_dvc
    setup_mlflow
    setup_airflow
    create_env_file
    display_next_steps
    
    log_success "All setup steps completed!"
}

# Run main function
main "$@"
