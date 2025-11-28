#!/bin/bash

################################################################################
# Cleanup Script
#
# Stops all services and cleans up temporary files
# Use with caution - this will stop all running services!
#
# Author: Amey Talkatkar
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

################################################################################
# Stop Running Services
################################################################################

stop_services() {
    print_header "Stopping Running Services"
    
    # Stop API server
    if [[ -f "/tmp/mlops-api.pid" ]]; then
        log_info "Stopping API server..."
        PID=$(cat /tmp/mlops-api.pid)
        if ps -p $PID > /dev/null; then
            kill $PID
            log_success "API server stopped (PID: $PID)"
        fi
        rm -f /tmp/mlops-api.pid
    else
        # Try to find and kill uvicorn processes
        if pgrep -f "uvicorn.*api.main:app" > /dev/null; then
            log_info "Killing uvicorn processes..."
            pkill -f "uvicorn.*api.main:app"
            log_success "Uvicorn processes killed"
        fi
    fi
    
    # Stop Dashboard
    if [[ -f "/tmp/mlops-dashboard.pid" ]]; then
        log_info "Stopping Dashboard..."
        PID=$(cat /tmp/mlops-dashboard.pid)
        if ps -p $PID > /dev/null; then
            kill $PID
            log_success "Dashboard stopped (PID: $PID)"
        fi
        rm -f /tmp/mlops-dashboard.pid
    else
        # Try to find and kill streamlit processes
        if pgrep -f "streamlit run.*dashboard" > /dev/null; then
            log_info "Killing streamlit processes..."
            pkill -f "streamlit run.*dashboard"
            log_success "Streamlit processes killed"
        fi
    fi
    
    # Stop MLflow server
    if pgrep -f "mlflow server" > /dev/null; then
        log_info "Stopping MLflow server..."
        pkill -f "mlflow server"
        log_success "MLflow server stopped"
    fi
    
    # Stop Airflow (if running standalone)
    if pgrep -f "airflow standalone" > /dev/null; then
        log_info "Stopping Airflow standalone..."
        pkill -f "airflow standalone"
        log_success "Airflow stopped"
    fi
    
    # Stop Airflow webserver
    if pgrep -f "airflow webserver" > /dev/null; then
        log_info "Stopping Airflow webserver..."
        pkill -f "airflow webserver"
    fi
    
    # Stop Airflow scheduler
    if pgrep -f "airflow scheduler" > /dev/null; then
        log_info "Stopping Airflow scheduler..."
        pkill -f "airflow scheduler"
    fi
    
    log_success "All services stopped"
}

################################################################################
# Stop Systemd Services
################################################################################

stop_systemd_services() {
    print_header "Stopping Systemd Services"
    
    services=(
        "mlops-api"
        "mlops-dashboard"
        "mlflow-server"
    )
    
    for service in "${services[@]}"; do
        if systemctl is-active --quiet ${service}; then
            log_info "Stopping ${service}..."
            sudo systemctl stop ${service}
            log_success "${service} stopped"
        else
            log_info "${service} is not running"
        fi
    done
}

################################################################################
# Clean Temporary Files
################################################################################

clean_temp_files() {
    print_header "Cleaning Temporary Files"
    
    # PID files
    log_info "Removing PID files..."
    rm -f /tmp/mlops-*.pid
    
    # Python cache
    log_info "Removing Python cache..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Pytest cache
    log_info "Removing pytest cache..."
    rm -rf .pytest_cache 2>/dev/null || true
    rm -f .coverage 2>/dev/null || true
    rm -rf htmlcov 2>/dev/null || true
    
    # Logs (optional - commented out by default)
    # log_info "Cleaning old logs..."
    # find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    log_success "Temporary files cleaned"
}

################################################################################
# Clean Docker Resources
################################################################################

clean_docker() {
    print_header "Cleaning Docker Resources"
    
    if ! command -v docker &> /dev/null; then
        log_info "Docker not installed, skipping..."
        return
    fi
    
    # Stop containers
    if docker ps -q --filter "name=mlops" | grep -q .; then
        log_info "Stopping MLOps containers..."
        docker stop $(docker ps -q --filter "name=mlops")
        log_success "Containers stopped"
    fi
    
    # Remove containers
    if docker ps -a -q --filter "name=mlops" | grep -q .; then
        log_info "Removing MLOps containers..."
        docker rm $(docker ps -a -q --filter "name=mlops")
        log_success "Containers removed"
    fi
    
    # Clean up images (optional - commented out by default)
    # log_info "Removing MLOps images..."
    # docker rmi $(docker images --filter "reference=mlops*" -q) 2>/dev/null || true
    
    # Clean up volumes (optional - commented out by default)
    # log_warning "Removing MLOps volumes (THIS WILL DELETE DATA)..."
    # docker volume rm $(docker volume ls -q --filter "name=mlops") 2>/dev/null || true
    
    log_success "Docker cleanup complete"
}

################################################################################
# Reset Databases (Optional)
################################################################################

reset_databases() {
    print_header "Database Reset (Optional)"
    
    echo ""
    log_warning "This will DELETE all data from the database!"
    read -p "Are you sure you want to reset databases? (yes/no): " -r
    echo
    
    if [[ $REPLY != "yes" ]]; then
        log_info "Database reset skipped"
        return
    fi
    
    log_info "Resetting PostgreSQL database..."
    
    sudo -u postgres psql <<EOF
-- Drop and recreate database
DROP DATABASE IF EXISTS airflow;
CREATE DATABASE airflow OWNER airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
EOF
    
    log_success "Database reset complete"
    log_warning "You will need to run 'airflow db migrate' again"
}

################################################################################
# Clean MLflow Artifacts
################################################################################

clean_mlflow_artifacts() {
    print_header "Cleaning MLflow Artifacts (Optional)"
    
    if [[ ! -d "mlruns" ]] && [[ ! -d "mlartifacts" ]]; then
        log_info "No MLflow artifacts found"
        return
    fi
    
    echo ""
    log_warning "This will DELETE all MLflow experiments and artifacts!"
    read -p "Clean MLflow artifacts? (yes/no): " -r
    echo
    
    if [[ $REPLY != "yes" ]]; then
        log_info "MLflow cleanup skipped"
        return
    fi
    
    log_info "Removing MLflow artifacts..."
    rm -rf mlruns 2>/dev/null || true
    rm -rf mlartifacts 2>/dev/null || true
    
    log_success "MLflow artifacts cleaned"
}

################################################################################
# Clean Airflow Logs
################################################################################

clean_airflow_logs() {
    print_header "Cleaning Airflow Logs (Optional)"
    
    if [[ ! -d "airflow/logs" ]]; then
        log_info "No Airflow logs found"
        return
    fi
    
    echo ""
    read -p "Clean Airflow logs? (yes/no): " -r
    echo
    
    if [[ $REPLY != "yes" ]]; then
        log_info "Airflow logs cleanup skipped"
        return
    fi
    
    log_info "Removing Airflow logs..."
    find airflow/logs -type f -name "*.log" -delete 2>/dev/null || true
    
    log_success "Airflow logs cleaned"
}

################################################################################
# Display System Status
################################################################################

display_status() {
    print_header "System Status"
    
    echo ""
    echo "📊 Process Status:"
    echo ""
    
    # Check for running processes
    if pgrep -f "uvicorn" > /dev/null; then
        echo "   🟢 API (uvicorn): RUNNING"
    else
        echo "   🔴 API (uvicorn): STOPPED"
    fi
    
    if pgrep -f "streamlit" > /dev/null; then
        echo "   🟢 Dashboard (streamlit): RUNNING"
    else
        echo "   🔴 Dashboard (streamlit): STOPPED"
    fi
    
    if pgrep -f "mlflow server" > /dev/null; then
        echo "   🟢 MLflow: RUNNING"
    else
        echo "   🔴 MLflow: STOPPED"
    fi
    
    if pgrep -f "airflow" > /dev/null; then
        echo "   🟢 Airflow: RUNNING"
    else
        echo "   🔴 Airflow: STOPPED"
    fi
    
    # Check systemd services
    echo ""
    echo "🔧 Systemd Services:"
    echo ""
    
    for service in mlops-api mlops-dashboard mlflow-server; do
        if systemctl is-active --quiet ${service} 2>/dev/null; then
            echo "   🟢 ${service}: ACTIVE"
        elif systemctl is-enabled --quiet ${service} 2>/dev/null; then
            echo "   🟡 ${service}: ENABLED (not running)"
        else
            echo "   🔴 ${service}: INACTIVE"
        fi
    done
    
    echo ""
}

################################################################################
# Main Menu
################################################################################

show_menu() {
    clear
    print_header "MLOps Pipeline Cleanup"
    
    echo ""
    echo "Select cleanup option:"
    echo ""
    echo "1) Stop all services"
    echo "2) Stop systemd services"
    echo "3) Clean temporary files"
    echo "4) Clean Docker resources"
    echo "5) Reset databases (DANGEROUS)"
    echo "6) Clean MLflow artifacts"
    echo "7) Clean Airflow logs"
    echo "8) Full cleanup (1-4)"
    echo "9) Display system status"
    echo "0) Exit"
    echo ""
}

################################################################################
# Main Execution
################################################################################

main() {
    # Check if running with sudo for systemd operations
    if [[ $EUID -ne 0 ]] && [[ "$1" == "systemd" ]]; then
        log_warning "Systemd operations require sudo"
        exec sudo "$0" "$@"
    fi
    
    # If no arguments, show interactive menu
    if [[ $# -eq 0 ]]; then
        while true; do
            show_menu
            read -p "Enter choice [0-9]: " choice
            
            case $choice in
                1) stop_services ;;
                2) stop_systemd_services ;;
                3) clean_temp_files ;;
                4) clean_docker ;;
                5) reset_databases ;;
                6) clean_mlflow_artifacts ;;
                7) clean_airflow_logs ;;
                8) 
                    stop_services
                    clean_temp_files
                    clean_docker
                    ;;
                9) display_status ;;
                0) 
                    log_info "Exiting..."
                    exit 0
                    ;;
                *) 
                    log_error "Invalid option!"
                    sleep 2
                    ;;
            esac
            
            echo ""
            read -p "Press Enter to continue..."
        done
    fi
    
    # Command-line arguments
    case "$1" in
        stop)
            stop_services
            ;;
        systemd)
            stop_systemd_services
            ;;
        temp)
            clean_temp_files
            ;;
        docker)
            clean_docker
            ;;
        all)
            stop_services
            stop_systemd_services
            clean_temp_files
            clean_docker
            ;;
        status)
            display_status
            ;;
        *)
            echo "Usage: $0 {stop|systemd|temp|docker|all|status}"
            echo ""
            echo "Or run without arguments for interactive menu"
            exit 1
            ;;
    esac
}

# Run main
main "$@"
